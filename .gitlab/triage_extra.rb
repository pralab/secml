module TriageExtra

  def has_version_label?
    version_labels.any?
  end

  def milestone_open?
    !milestone&.closed?
  end

  def has_milestone_label?
    labels.include?(milestone&.title)
  end

  def no_milestone_labels
    version_labels - [milestone&.title]
  end

  def unlabel_no_milestone_labels
    no_milestone_labels.empty? ? "" : "/unlabel " + no_milestone_labels.map { |i| "~\"" + i + "\"" }.join(" ")
  end

  def labels
    resource[:labels]
  end

  def version_labels
    labels.grep(/^v\d+\..*$/)
  end

end

Gitlab::Triage::Resource::Context.include TriageExtra
